/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use euclid::default::Size2D;

use fnv::FnvHashMap;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::mem;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;

use surfman::Context;
use surfman::Error;
use surfman::Device;
use surfman::Surface;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SwapChainId(pub usize);

impl Display for SwapChainId {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", *self)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct ContextId(usize);

impl<'a> From<&'a mut Context> for ContextId {
    fn from(context: &'a mut Context) -> ContextId {
        // TODO: context ids shouldn't just be addresses
	ContextId(context as *const Context as usize)
    }
}

struct SwapChainData {
    id: SwapChainId,
    size: Size2D<i32>,
    context_id: ContextId,
    unattached_front_buffer: Option<Surface>,
    pending_surface: Option<Surface>,
    presented_surfaces: Vec<Surface>,
}

impl SwapChainData {
    fn validate_context(&self, context: &mut Context) -> Result<(), Error> {
        if self.context_id == ContextId::from(context) {
	    Ok(())
	} else {
  	    Err(Error::IncompatibleContext)
	}
    }

    fn swap_buffers(&mut self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.validate_context(context)?;

        // Fetch a new back buffer, recycling presented buffers if possible.
	let new_back_buffer = self.presented_surfaces
	    .iter()
	    .position(|surface| surface.size() == self.size)
	    .map(|index| Ok(self.presented_surfaces.swap_remove(index)))
	    .unwrap_or_else(|| device.create_surface(context, &self.size))?;

        // Swap the buffers
	let new_front_buffer = match self.unattached_front_buffer.as_mut() {
	    Some(surface) => mem::replace(surface, new_back_buffer),
	    None => device.replace_context_surface(context, new_back_buffer)?,
        };

        // Updata the state
	self.pending_surface = Some(new_front_buffer);
	for surface in self.presented_surfaces.drain(..) {
	    device.destroy_surface(context, surface)?;
	}

	Ok(())
    }

    fn take_surface(&mut self) -> Option<Surface> {
        self.pending_surface.take().or_else(|| self.presented_surfaces.pop())
    }

    fn recycle_surface(&mut self, surface: Surface) {
        self.presented_surfaces.push(surface)
    }
}

#[derive(Clone)]
pub struct SwapChain(Arc<Mutex<SwapChainData>>);

impl SwapChain {
    fn lock(&self) -> MutexGuard<SwapChainData> {
        self.0.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn id(&self) -> SwapChainId {
        self.lock().id
    }

    pub fn swap_buffers(&self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.lock().swap_buffers(device, context)
    }

    pub fn take_surface(&self) -> Option<Surface> {
        self.lock().take_surface()
    }

    pub fn recycle_surface(&self, surface: Surface) {
        self.lock().recycle_surface(surface)
    }
}

#[derive(Clone, Default)]
pub struct SwapChains {
    next_id: Arc<AtomicUsize>,
    table: Arc<RwLock<FnvHashMap<SwapChainId, SwapChain>>>,
}

impl SwapChains {
    fn table(&self) -> RwLockReadGuard<FnvHashMap<SwapChainId, SwapChain>> {
        self.table.read().unwrap_or_else(|err| err.into_inner())
    }

    fn table_mut(&self) -> RwLockWriteGuard<FnvHashMap<SwapChainId, SwapChain>> {
        self.table.write().unwrap_or_else(|err| err.into_inner())
    }

    pub fn get(&self, id: SwapChainId) -> Option<SwapChain> {
        self.table().get(&id).cloned()
    }

    pub fn get_with<F, T>(&self, id: SwapChainId, f: F) -> Option<T> where
        F: Fn(&SwapChain) -> T,
    {
        self.table().get(&id).map(f)
    }

    pub fn create_swap_chain(&self, context: &mut Context, size: Size2D<i32>) -> SwapChainId {
        let id = SwapChainId(self.next_id.fetch_add(1, Ordering::SeqCst));
	self.table_mut().entry(id).or_insert_with(move || {
            SwapChain(Arc::new(Mutex::new(SwapChainData {
	        id,
	        size,
	        context_id: ContextId::from(context),
                unattached_front_buffer: None,
                pending_surface: None,
                presented_surfaces: Vec::new(),
	    })))
	});
	id
    }
}